import os
import traceback
import warnings
from os.path import join
from stat import ST_MTIME
import re
import runpy
from docutils import nodes
from docutils.parsers.rst.roles import set_classes
from subprocess import check_call, DEVNULL, CalledProcessError
from pathlib import Path
import matplotlib
def create_png_files(raise_exceptions=False):
    from ase.utils import workdir
    try:
        check_call(['povray', '-h'], stderr=DEVNULL)
    except (FileNotFoundError, CalledProcessError):
        warnings.warn('No POVRAY!')
        from ase.io import pov
        from ase.io.png import write_png

        def write_pov(filename, atoms, povray_settings={}, isosurface_data=None, **generic_projection_settings):
            write_png(Path(filename).with_suffix('.png'), atoms, **generic_projection_settings)

            class DummyRenderer:

                def render(self):
                    pass
            return DummyRenderer()
        pov.write_pov = write_pov
    for dir, pyname, outnames in creates():
        path = join(dir, pyname)
        t0 = os.stat(path)[ST_MTIME]
        run = False
        for outname in outnames:
            try:
                t = os.stat(join(dir, outname))[ST_MTIME]
            except OSError:
                run = True
                break
            else:
                if t < t0:
                    run = True
                    break
        if run:
            print('running:', path)
            with workdir(dir):
                import matplotlib.pyplot as plt
                plt.figure()
                try:
                    runpy.run_path(pyname)
                except KeyboardInterrupt:
                    return
                except Exception:
                    if raise_exceptions:
                        raise
                    else:
                        traceback.print_exc()
            for n in plt.get_fignums():
                plt.close(n)
            for outname in outnames:
                print(dir, outname)