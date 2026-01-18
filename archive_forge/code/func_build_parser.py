import os
import subprocess
import glob
from sympy.utilities.misc import debug
def build_parser(output_dir=dir_autolev_antlr):
    check_antlr_version()
    debug('Updating ANTLR-generated code in {}'.format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, '__init__.py'), 'w+') as fp:
        fp.write(header)
    args = ['antlr4', grammar_file, '-o', output_dir, '-no-visitor']
    debug('Running code generation...\n\t$ {}'.format(' '.join(args)))
    subprocess.check_output(args, cwd=output_dir)
    debug('Applying headers, removing unnecessary files and renaming...')
    for path in glob.glob(os.path.join(output_dir, 'Autolev*.*')) or glob.glob(os.path.join(output_dir, 'autolev*.*')):
        if not path.endswith('.py'):
            os.unlink(path)
            continue
        new_path = os.path.join(output_dir, os.path.basename(path).lower())
        with open(path, 'r') as f:
            lines = [line.rstrip().replace('AutolevParser import', 'autolevparser import') + '\n' for line in f.readlines()]
        os.unlink(path)
        with open(new_path, 'w') as out_file:
            offset = 0
            while lines[offset].startswith('#'):
                offset += 1
            out_file.write(header)
            out_file.writelines(lines[offset:])
        debug('\t{}'.format(new_path))
    return True