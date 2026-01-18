import re
from io import BytesIO
from .lazy_import import lazy_import
from fnmatch import fnmatch
from breezy._termcolor import color_string, FG
from breezy import (
from . import controldir, errors, osutils
from . import revision as _mod_revision
from . import trace
from .revisionspec import RevisionSpec, RevisionSpec_revid, RevisionSpec_revno
def grep_diff(opts):
    wt, branch, relpath = controldir.ControlDir.open_containing_tree_or_branch('.')
    with branch.lock_read():
        if opts.revision:
            start_rev = opts.revision[0]
        else:
            opts.revision = [RevisionSpec.from_string('revno:1'), RevisionSpec.from_string('last:1')]
            start_rev = opts.revision[0]
        start_revid = start_rev.as_revision_id(branch)
        if start_revid == b'null:':
            return
        srevno_tuple = branch.revision_id_to_dotted_revno(start_revid)
        if len(opts.revision) == 2:
            end_rev = opts.revision[1]
            end_revid = end_rev.as_revision_id(branch)
            if end_revid is None:
                end_revno, end_revid = branch.last_revision_info()
            erevno_tuple = branch.revision_id_to_dotted_revno(end_revid)
            grep_mainline = _rev_on_mainline(srevno_tuple) and _rev_on_mainline(erevno_tuple)
            if srevno_tuple > erevno_tuple:
                srevno_tuple, erevno_tuple = (erevno_tuple, srevno_tuple)
                start_revid, end_revid = (end_revid, start_revid)
            if opts.levels == 1 and grep_mainline:
                given_revs = _linear_view_revisions(branch, start_revid, end_revid)
            else:
                given_revs = _graph_view_revisions(branch, start_revid, end_revid)
        else:
            start_revno = '.'.join(map(str, srevno_tuple))
            start_rev_tuple = (start_revid, start_revno, 0)
            given_revs = [start_rev_tuple]
        repo = branch.repository
        diff_pattern = re.compile(b'^[+\\-].*(' + opts.pattern.encode(_user_encoding) + b')')
        file_pattern = re.compile(b"=== (modified|added|removed) file '.*'")
        outputter = _GrepDiffOutputter(opts)
        writeline = outputter.get_writer()
        writerevno = outputter.get_revision_header_writer()
        writefileheader = outputter.get_file_header_writer()
        file_encoding = _user_encoding
        for revid, revno, merge_depth in given_revs:
            if opts.levels == 1 and merge_depth != 0:
                continue
            rev_spec = RevisionSpec_revid.from_string('revid:' + revid.decode('utf-8'))
            new_rev = repo.get_revision(revid)
            new_tree = rev_spec.as_tree(branch)
            if len(new_rev.parent_ids) == 0:
                ancestor_id = _mod_revision.NULL_REVISION
            else:
                ancestor_id = new_rev.parent_ids[0]
            old_tree = repo.revision_tree(ancestor_id)
            s = BytesIO()
            diff.show_diff_trees(old_tree, new_tree, s, old_label='', new_label='')
            display_revno = True
            display_file = False
            file_header = None
            text = s.getvalue()
            for line in text.splitlines():
                if file_pattern.search(line):
                    file_header = line
                    display_file = True
                elif diff_pattern.search(line):
                    if display_revno:
                        writerevno('=== revno:{} ==='.format(revno))
                        display_revno = False
                    if display_file:
                        writefileheader('  {}'.format(file_header.decode(file_encoding, 'replace')))
                        display_file = False
                    line = line.decode(file_encoding, 'replace')
                    writeline('    {}'.format(line))