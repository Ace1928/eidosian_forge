from __future__ import print_function
import os
import platform
import remove_pyreadline
import setuptools.command.easy_install as easy_install
import setuptools.package_index
import shutil
import sys
def delete_pyreadline():
    pkg = locate_package('pyreadline')
    if pkg is None:
        print('pyreadline not found, exiting.')
        return
    consumers = find_package_consumers('pyreadline')
    if consumers:
        print('pyreadline is a dependency of all the following packages:')
        for p in consumers:
            print('  %s' % (p,))
        print()
    else:
        print('pyreadline is not a dependency of any installed packages.')
        print()
    response = y_or_n_p('Continue and uninstall pyreadline?')
    if response == 'n':
        print('Aborting uninstall of pyreadline.')
        return
    remove_package(pkg)
    print('pyreadline successfully uninstalled!')