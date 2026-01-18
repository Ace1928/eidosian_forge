import re
import sys
import os.path as op
from glob import glob
from traits.trait_errors import TraitError
from nipype.utils.filemanip import loadcrash
def display_crash_file(crashfile, rerun, debug, directory):
    """display crash file content and rerun if required"""
    from nipype.utils.filemanip import loadcrash
    crash_data = loadcrash(crashfile)
    node = None
    if 'node' in crash_data:
        node = crash_data['node']
    tb = crash_data['traceback']
    print('\n')
    print('File: %s' % crashfile)
    if node:
        print('Node: %s' % node)
        if node.base_dir:
            print('Working directory: %s' % node.output_dir())
        else:
            print('Node crashed before execution')
        print('\n')
        print('Node inputs:')
        print(node.inputs)
        print('\n')
    print('Traceback: ')
    print(''.join(tb))
    print('\n')
    if rerun:
        if node is None:
            print('No node in crashfile. Cannot rerun')
            return
        print('Rerunning node')
        node.base_dir = directory
        node.config = {'execution': {'crashdump_dir': '/tmp'}}
        try:
            node.run()
        except:
            if debug and debug != 'ipython':
                import pdb
                pdb.post_mortem()
            else:
                raise
        print('\n')