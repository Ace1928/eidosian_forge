import platform
import sys
from ase.dependencies import all_dependencies
from ase.io.formats import filetype, ioformats, UnknownFileTypeError
from ase.io.ulm import print_ulm_info
from ase.io.bundletrajectory import print_bundletrajectory_info
def print_formats():
    print('Supported formats:')
    for fmtname in sorted(ioformats):
        fmt = ioformats[fmtname]
        infos = [fmt.modes, 'single' if fmt.single else 'multi']
        if fmt.isbinary:
            infos.append('binary')
        if fmt.encoding is not None:
            infos.append(fmt.encoding)
        infostring = '/'.join(infos)
        moreinfo = [infostring]
        if fmt.extensions:
            moreinfo.append('ext={}'.format('|'.join(fmt.extensions)))
        if fmt.globs:
            moreinfo.append('glob={}'.format('|'.join(fmt.globs)))
        print('  {} [{}]: {}'.format(fmt.name, ', '.join(moreinfo), fmt.description))