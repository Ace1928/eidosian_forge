from nipype.interfaces import fsl
from nipype.utils import docparse
import subprocess
from ..interfaces.base import CommandLine
from .misc import is_container
def get_params_from_doc(cmd, style='--', help_flag=None, trap_error=True):
    """Auto-generate option map from command line help

    Parameters
    ----------
    cmd : string
        The command whose documentation we are fetching
    style : string default ['--']
        The help command style (--, -). Multiple styles can be provided in a
        list e.g. ['--','-'].
    help_flag : string
        Provide additional help flag. e.g., -h
    trap_error : boolean
        Override if underlying command returns a non-zero returncode

    Returns
    -------
    optmap : dict
        Contains a mapping from input to command line variables

    """
    res = CommandLine('which %s' % cmd.split(' ')[0], resource_monitor=False, terminal_output='allatonce').run()
    cmd_path = res.runtime.stdout.strip()
    if cmd_path == '':
        raise Exception('Command %s not found' % cmd.split(' ')[0])
    if help_flag:
        cmd = ' '.join((cmd, help_flag))
    doc = grab_doc(cmd, trap_error)
    return _parse_doc(doc, style)