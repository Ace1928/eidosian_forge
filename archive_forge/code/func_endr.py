import enum
import logging
import os
import sys
import typing
import warnings
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import callbacks
def endr(fatal: int) -> None:
    logger.debug('Ending embedded R process.')
    global rpy2_embeddedR_isinitialized
    rlib = openrlib.rlib
    with openrlib.rlock:
        if rpy2_embeddedR_isinitialized & RPY_R_Status.ENDED.value:
            logger.info('Embedded R already ended.')
            return
        logger.debug('R_do_Last()')
        rlib.R_dot_Last()
        logger.debug('R_RunExitFinalizers()')
        rlib.R_RunExitFinalizers()
        logger.debug('Rf_KillAllDevices()')
        rlib.Rf_KillAllDevices()
        logger.debug('R_CleanTempDir()')
        rlib.R_CleanTempDir()
        logger.debug('R_gc')
        rlib.R_gc()
        logger.debug('Rf_endEmbeddedR(fatal)')
        rlib.Rf_endEmbeddedR(fatal)
        rpy2_embeddedR_isinitialized ^= RPY_R_Status.ENDED.value
        logger.info('Embedded R ended.')