from __future__ import annotations
import glob
import os
import re
import time
from typing import Optional
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from os_brick import constants
from os_brick import exception
from os_brick import executor
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
@staticmethod
def lun_for_addressing(lun, addressing_mode=None):
    """Convert luns to values used by the system.

        How a LUN is codified depends on the standard being used by the storage
        array and the mode, which is unknown by the host.

        Addressing modes based on the standard:
            * SAM:
              - 64bit address

            * SAM-2:
              - Peripheral device addressing method (Code 00b)
                + Single level
                + Multi level
              - Flat space addressing method (Code 01b)
              - Logical unit addressing mode (Code 10b)
              - Extended logical unit addressing method (Code 11b)

            * SAM-3: Mostly same as SAM-2 but with some differences,
              like supporting addressing LUNs < 256 with flat address space.

        This means that the same LUN numbers could have different addressing
        values.  Examples:
          * LUN 1:
            - SAM representation: 1
            - SAM-2 peripheral: 1
            - SAM-2 flat addressing: Invalid
            - SAM-3 flat addressing: 16384

          * LUN 256
            - SAM representation: 256
            - SAM-2 peripheral: Not possible to represent
            - SAM-2 flat addressing: 16640
            - SAM-3 flat addressing: 16640

        This method makes the transformation from the numerical LUN value to
        the right addressing value based on the addressing_mode.

        Acceptable values are:
        - SAM: 64bit address with no translation
        - transparent: Same as SAM but used by drivers that want to use non
                       supported addressing modes by using the addressing mode
                       instead of the LUN without being misleading (untested).
        - SAM2: Peripheral for LUN < 256 and flat for LUN >= 256. In SAM-2
                flat cannot be used for 0-255
        - SAM3-flat: Force flat-space addressing

        The default is SAM/transparent and nothing will be done with the LUNs.
        """
    mode = addressing_mode or constants.SCSI_ADDRESSING_SAM
    if mode not in constants.SCSI_ADDRESSING_MODES:
        raise exception.InvalidParameterValue(f'Invalid addressing_mode {addressing_mode}')
    if mode == constants.SCSI_ADDRESSING_SAM3_FLAT or (mode == constants.SCSI_ADDRESSING_SAM2 and lun >= 256):
        old_lun = lun
        lun += 16384
        LOG.info('Transforming LUN value for addressing: %s -> %s', old_lun, lun)
    return lun