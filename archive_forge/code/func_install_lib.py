import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
def install_lib(cuda, prefix, library):
    if platform.uname().machine.lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('\nCurrently this tool only supports x86_64 architecture.')
    record = None
    lib_records = library_records
    for record in lib_records[library]:
        if record['cuda'] == cuda:
            break
    else:
        raise RuntimeError('\nThe CUDA version specified is not supported.\nShould be one of {}.'.format(str([x['cuda'] for x in lib_records[library]])))
    if prefix is None:
        prefix = os.path.expanduser('~/.cupy/cuda_lib')
    destination = calculate_destination(prefix, cuda, library, record[library])
    if os.path.exists(destination):
        raise RuntimeError('\nThe destination directory {} already exists.\nRemove the directory first if you want to reinstall.'.format(destination))
    target_platform = platform.system()
    asset = record['assets'].get(target_platform, None)
    if asset is None:
        raise RuntimeError('\nThe current platform ({}) is not supported.'.format(target_platform))
    if library == 'cudnn':
        print('By downloading and using cuDNN, you accept the terms and conditions of the NVIDIA cuDNN Software License Agreement:')
        print('  https://docs.nvidia.com/deeplearning/cudnn/sla/index.html')
        print()
    elif library == 'cutensor':
        print('By downloading and using cuTENSOR, you accept the terms and conditions of the NVIDIA cuTENSOR Software License Agreement:')
        print('  https://docs.nvidia.com/cuda/cutensor/license.html')
        print()
    elif library == 'nccl':
        pass
    else:
        assert False
    print('Installing {} {} for CUDA {} to: {}'.format(library, record[library], record['cuda'], destination))
    url = asset['url']
    print('Downloading {}...'.format(url))
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, os.path.basename(url)), 'wb') as f:
            with urllib.request.urlopen(url) as response:
                f.write(response.read())
        print('Extracting...')
        outdir = os.path.join(tmpdir, 'extract')
        _unpack_archive(f.name, outdir)
        subdir = os.listdir(outdir)
        assert len(subdir) == 1
        dir_name = subdir[0]
        print('Installing...')
        if library == 'cudnn':
            libdirs = ['bin', 'lib'] if sys.platform == 'win32' else ['lib']
            for item in libdirs + ['include', 'LICENSE']:
                shutil.move(os.path.join(outdir, dir_name, item), os.path.join(destination, item))
        elif library == 'cutensor':
            if cuda.startswith('11.') and cuda != '11.0':
                cuda = '11'
            elif cuda.startswith('12.'):
                cuda = '12'
            license = 'LICENSE'
            shutil.move(os.path.join(outdir, dir_name, 'include'), os.path.join(destination, 'include'))
            shutil.move(os.path.join(outdir, dir_name, 'lib', cuda), os.path.join(destination, 'lib'))
            shutil.move(os.path.join(outdir, dir_name, license), destination)
        elif library == 'nccl':
            shutil.move(os.path.join(outdir, dir_name), destination)
        else:
            assert False
        print('Cleaning up...')
    print('Done!')