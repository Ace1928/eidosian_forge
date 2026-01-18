import os
def raise_build_error(e):
    local_dir = os.path.split(__file__)[0]
    msg = STANDARD_MSG
    if local_dir == 'sklearn/__check_build':
        msg = INPLACE_MSG
    dir_content = list()
    for i, filename in enumerate(os.listdir(local_dir)):
        if (i + 1) % 3:
            dir_content.append(filename.ljust(26))
        else:
            dir_content.append(filename + '\n')
    raise ImportError('%s\n___________________________________________________________________________\nContents of %s:\n%s\n___________________________________________________________________________\nIt seems that scikit-learn has not been built correctly.\n\nIf you have installed scikit-learn from source, please do not forget\nto build the package before using it: run `python setup.py install` or\n`make` in the source directory.\n%s' % (e, local_dir, ''.join(dir_content).strip(), msg))