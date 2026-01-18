import os
import subprocess
import shutil
import tempfile
def rsys2graph(rsys, fname, output_dir=None, prog=None, save=False, **kwargs):
    """
    Convenience function to call `rsys2dot` and write output to file
    and render the graph

    Parameters
    ----------
    rsys : ReactionSystem
    fname : str
        filename
    output_dir : str (optional)
        path to directory (default: temporary directory)
    prog : str (optional)
        default: 'dot'
    save : bool
        removes temporary directory if False, default: False
    \\*\\*kwargs :
        Keyword arguments passed along to py:func:`rsys2dot`.

    Returns
    -------
    str
        Outpath

    Examples
    --------
    >>> rsys2graph(rsys, sbstncs, '/tmp/out.png')  # doctest: +SKIP

    """
    lines = rsys2dot(rsys, **kwargs)
    created_tempdir = False
    try:
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
            created_tempdir = True
        basename, ext = os.path.splitext(os.path.basename(fname))
        outpath = os.path.join(output_dir, fname)
        dotpath = os.path.join(output_dir, basename + '.dot')
        with open(dotpath, 'wt') as ofh:
            ofh.writelines(lines)
        if ext == '.tex':
            cmds = [prog or 'dot2tex']
        else:
            cmds = [prog or 'dot', '-T' + outpath.split('.')[-1]]
        p = subprocess.Popen(cmds + [dotpath, '-o', outpath])
        retcode = p.wait()
        if retcode:
            fmtstr = '{}\n returned with exit status {}'
            raise RuntimeError(fmtstr.format(' '.join(cmds), retcode))
        return outpath
    finally:
        if save is True or save == 'True':
            pass
        elif save is False or save == 'False':
            if created_tempdir:
                shutil.rmtree(output_dir)
        else:
            shutil.copy(outpath, save)