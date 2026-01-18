import os
import getpass
from socket import gethostname
import sys
import uuid
from time import strftime
from traceback import format_exception
from ... import logging
from ...utils.filemanip import savepkl, crash2txt
import sys
import os
from nipype import config, logging
from nipype.utils.filemanip import loadpkl, savepkl
from socket import gethostname
from traceback import format_exception
from nipype.utils.filemanip import loadpkl, savepkl
def report_nodes_not_run(notrun):
    """List nodes that crashed with crashfile info

    Optionally displays dependent nodes that weren't executed as a result of
    the crash.
    """
    if notrun:
        logger.info('***********************************')
        for info in notrun:
            logger.error('could not run node: %s' % '.'.join((info['node']._hierarchy, info['node']._id)))
            logger.info('crashfile: %s' % info['crashfile'])
            logger.debug('The following dependent nodes were not run')
            for subnode in info['dependents']:
                logger.debug(subnode._id)
        logger.info('***********************************')