from __future__ import absolute_import, division, print_function
import io
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_text
def do_ini(module, filename, section=None, option=None, values=None, state='present', exclusive=True, backup=False, no_extra_spaces=False, ignore_spaces=False, create=True, allow_no_value=False, modify_inactive_option=True, follow=False):
    if section is not None:
        section = to_text(section)
    if option is not None:
        option = to_text(option)
    values_unique = []
    [values_unique.append(to_text(value)) for value in values if value not in values_unique and value is not None]
    values = values_unique
    diff = dict(before='', after='', before_header='%s (content)' % filename, after_header='%s (content)' % filename)
    if follow and os.path.islink(filename):
        target_filename = os.path.realpath(filename)
    else:
        target_filename = filename
    if not os.path.exists(target_filename):
        if not create:
            module.fail_json(rc=257, msg='Destination %s does not exist!' % target_filename)
        destpath = os.path.dirname(target_filename)
        if not os.path.exists(destpath) and (not module.check_mode):
            os.makedirs(destpath)
        ini_lines = []
    else:
        with io.open(target_filename, 'r', encoding='utf-8-sig') as ini_file:
            ini_lines = [to_text(line) for line in ini_file.readlines()]
    if module._diff:
        diff['before'] = u''.join(ini_lines)
    changed = False
    if not ini_lines:
        ini_lines.append(u'\n')
    if ini_lines[-1] == u'' or ini_lines[-1][-1] != u'\n':
        ini_lines[-1] += u'\n'
        changed = True
    fake_section_name = u'ad01e11446efb704fcdbdb21f2c43757423d91c5'
    ini_lines.insert(0, u'[%s]' % fake_section_name)
    ini_lines.append(u'[')
    if not section:
        section = fake_section_name
    within_section = not section
    section_start = section_end = 0
    msg = 'OK'
    if no_extra_spaces:
        assignment_format = u'%s=%s\n'
    else:
        assignment_format = u'%s = %s\n'
    option_no_value_present = False
    non_blank_non_comment_pattern = re.compile(to_text('^[ \\t]*([#;].*)?$'))
    before = after = []
    section_lines = []
    section_pattern = re.compile(to_text('^\\[\\s*%s\\s*]' % re.escape(section.strip())))
    for index, line in enumerate(ini_lines):
        if section_pattern.match(line):
            within_section = True
            section_start = index
        elif line.startswith(u'['):
            if within_section:
                section_end = index
                break
    before = ini_lines[0:section_start]
    section_lines = ini_lines[section_start:section_end]
    after = ini_lines[section_end:len(ini_lines)]
    changed_lines = [0] * len(section_lines)
    if modify_inactive_option:
        match_function = match_opt
    else:
        match_function = match_active_opt
    if state == 'present' and option:
        for index, line in enumerate(section_lines):
            if match_function(option, line):
                match = match_function(option, line)
                if values and match.group(7) in values:
                    matched_value = match.group(7)
                    if not matched_value and allow_no_value:
                        newline = u'%s\n' % option
                        option_no_value_present = True
                    else:
                        newline = assignment_format % (option, matched_value)
                    changed, msg = update_section_line(option, changed, section_lines, index, changed_lines, ignore_spaces, newline, msg)
                    values.remove(matched_value)
                elif not values and allow_no_value:
                    newline = u'%s\n' % option
                    changed, msg = update_section_line(option, changed, section_lines, index, changed_lines, ignore_spaces, newline, msg)
                    option_no_value_present = True
                    break
    if state == 'present' and exclusive and (not allow_no_value):
        if len(values) > 0:
            for index, line in enumerate(section_lines):
                if not changed_lines[index] and match_function(option, line):
                    newline = assignment_format % (option, values.pop(0))
                    changed, msg = update_section_line(option, changed, section_lines, index, changed_lines, ignore_spaces, newline, msg)
                    if len(values) == 0:
                        break
        for index in range(len(section_lines) - 1, 0, -1):
            if not changed_lines[index] and match_function(option, section_lines[index]):
                del section_lines[index]
                del changed_lines[index]
                changed = True
                msg = 'option changed'
    if state == 'present':
        for index in range(len(section_lines), 0, -1):
            if not non_blank_non_comment_pattern.match(section_lines[index - 1]):
                if option and values:
                    for element in values[::-1]:
                        if element is not None:
                            section_lines.insert(index, assignment_format % (option, element))
                            msg = 'option added'
                            changed = True
                        elif element is None and allow_no_value:
                            section_lines.insert(index, u'%s\n' % option)
                            msg = 'option added'
                            changed = True
                elif option and (not values) and allow_no_value and (not option_no_value_present):
                    section_lines.insert(index, u'%s\n' % option)
                    msg = 'option added'
                    changed = True
                break
    if state == 'absent':
        if option:
            if exclusive:
                new_section_lines = [line for line in section_lines if not match_active_opt(option, line)]
                if section_lines != new_section_lines:
                    changed = True
                    msg = 'option changed'
                    section_lines = new_section_lines
            elif not exclusive and len(values) > 0:
                new_section_lines = [i for i in section_lines if not (match_active_opt(option, i) and match_active_opt(option, i).group(7) in values)]
                if section_lines != new_section_lines:
                    changed = True
                    msg = 'option changed'
                    section_lines = new_section_lines
        elif section_lines:
            section_lines = []
            msg = 'section removed'
            changed = True
    ini_lines = before + section_lines + after
    del ini_lines[0]
    del ini_lines[-1:]
    if not within_section and state == 'present':
        ini_lines.append(u'[%s]\n' % section)
        msg = 'section and option added'
        if option and values:
            for value in values:
                ini_lines.append(assignment_format % (option, value))
        elif option and (not values) and allow_no_value:
            ini_lines.append(u'%s\n' % option)
        else:
            msg = 'only section added'
        changed = True
    if module._diff:
        diff['after'] = u''.join(ini_lines)
    backup_file = None
    if changed and (not module.check_mode):
        if backup:
            backup_file = module.backup_local(target_filename)
        encoded_ini_lines = [to_bytes(line) for line in ini_lines]
        try:
            tmpfd, tmpfile = tempfile.mkstemp(dir=module.tmpdir)
            f = os.fdopen(tmpfd, 'wb')
            f.writelines(encoded_ini_lines)
            f.close()
        except IOError:
            module.fail_json(msg='Unable to create temporary file %s', traceback=traceback.format_exc())
        try:
            module.atomic_move(tmpfile, target_filename)
        except IOError:
            module.ansible.fail_json(msg='Unable to move temporary                                    file %s to %s, IOError' % (tmpfile, target_filename), traceback=traceback.format_exc())
    return (changed, backup_file, diff, msg)