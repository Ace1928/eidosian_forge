import os
import argparse
def get_cache_file_set(args):
    """Get the list of files to be cached.

    Parameters
    ----------
    args: ArgumentParser.Argument
        The arguments returned by the parser.

    Returns
    -------
    cache_file_set: set of str
        The set of files to be cached to local execution environment.

    command: list of str
        The commands that get rewritten after the file cache is used.
    """
    fset = set()
    cmds = []
    if args.auto_file_cache:
        for i in range(len(args.command)):
            fname = args.command[i]
            if os.path.exists(fname):
                fset.add(fname)
                cmds.append('./' + fname.split('/')[-1])
            else:
                cmds.append(fname)
    for fname in args.files:
        if os.path.exists(fname):
            fset.add(fname)
    return (fset, cmds)