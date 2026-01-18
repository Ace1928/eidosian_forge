import os
import shutil
def rewrite_python_file_imports(target_dir, root_name, offset_name, verbose=False):
    """Rewrite internal package imports for move to a `src/` dir structure.

    If your project has its code in `package_name/`, and you want to instead
    move that code to `package_name/src/`, this script will change all lines
    of the form e.g.

    `from package_name.x.y import z` to `from package_name.src.x.y import z`

    if you call it as

    `rewrite_python_file_imports("package_name/", "package_name", "src")`
    """
    assert '.' not in offset_name
    for root, _, files in os.walk(target_dir):
        for fname in files:
            if fname.endswith('.py'):
                fpath = os.path.join(root, fname)
                if verbose:
                    print(f'...processing {fpath}')
                with open(fpath) as f:
                    contents = f.read()
                lines = contents.split('\n')
                in_string = False
                new_lines = []
                for line in lines:
                    if line.strip().startswith('"""') or line.strip().endswith('"""'):
                        if line.count('"') % 2 == 1:
                            in_string = not in_string
                    elif not in_string:
                        if line.strip() == f'import {root_name}':
                            line = line.replace(f'import {root_name}', f'import {root_name}.{offset_name} as {root_name}')
                        else:
                            line = line.replace(f'import {root_name}.', f'import {root_name}.{offset_name}.')
                            line = line.replace(f'from {root_name}.', f'from {root_name}.{offset_name}.')
                            line = line.replace(f'from {root_name} import', f'from {root_name}.{offset_name} import')
                    new_lines.append(line)
                with open(fpath, 'w') as f:
                    f.write('\n'.join(new_lines) + '\n')