import os
import shutil
import subprocess
import tempfile
from ..printing import latex
from ..kinetics.rates import RadiolyticBase
from ..units import to_unitless, get_derived_unit
def render_tex_to_pdf(contents, texfname, pdffname, output_dir, save):
    """Generates a pdf from a tex file by calling pdflatex

    Parameters
    ----------
    contents : str
    texfname : path
    pdffname : path
    output_dir : path
    save : path or bool or str(bool)

    """
    created_tempdir = False
    try:
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
            created_tempdir = True
        texpath = os.path.join(output_dir, texfname)
        pdfpath = os.path.join(output_dir, pdffname)
        cmds = ['pdflatex', '-halt-on-error', '-interaction', 'batchmode', texfname]
        with open(texpath, 'wt') as ofh:
            ofh.write(contents)
            ofh.flush()
        with open(pdfpath + '.out', 'wb') as logfile:
            p = subprocess.Popen(cmds, cwd=output_dir, stdout=logfile, stderr=logfile)
            retcode = p.wait()
            p = subprocess.Popen(cmds, cwd=output_dir, stdout=logfile, stderr=logfile)
            retcode += p.wait()
        if retcode:
            fmtstr = '{}\n returned with exit status {}'
            raise RuntimeError(fmtstr.format(' '.join(cmds), retcode))
        else:
            return pdfpath
    finally:
        if save is True or save == 'True':
            pass
        elif save is False or save == 'False':
            if created_tempdir:
                shutil.rmtree(output_dir)
        elif not os.path.samefile(pdfpath, save):
            shutil.copy(pdfpath, save)