import operator
from ... import (branch, commands, config, errors, option, trace, tsort, ui,
from ...revision import NULL_REVISION
from .classify import classify_delta
def display_info(info, to_file, gather_class_stats=None):
    """Write out the information"""
    for count, revs, emails, fullnames in info:
        sorted_emails = sorted(((count, email) for email, count in emails.items()), reverse=True)
        sorted_fullnames = sorted(((count, fullname) for fullname, count in fullnames.items()), reverse=True)
        if sorted_fullnames[0][1] == '' and sorted_emails[0][1] == '':
            to_file.write('%4d %s\n' % (count, 'Unknown'))
        else:
            to_file.write('%4d %s <%s>\n' % (count, sorted_fullnames[0][1], sorted_emails[0][1]))
        if len(sorted_fullnames) > 1:
            to_file.write('     Other names:\n')
            for count, fname in sorted_fullnames:
                to_file.write('     %4d ' % (count,))
                if fname == '':
                    to_file.write("''\n")
                else:
                    to_file.write('{}\n'.format(fname))
        if len(sorted_emails) > 1:
            to_file.write('     Other email addresses:\n')
            for count, email in sorted_emails:
                to_file.write('     %4d ' % (count,))
                if email == '':
                    to_file.write("''\n")
                else:
                    to_file.write('{}\n'.format(email))
        if gather_class_stats is not None:
            to_file.write('     Contributions:\n')
            classes, total = gather_class_stats(revs)
            for name, count in sorted(classes.items(), key=classify_key):
                if name is None:
                    name = 'Unknown'
                to_file.write('     %4.0f%% %s\n' % (float(count) / total * 100.0, name))