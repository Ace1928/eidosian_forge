from __future__ import annotations
import math
import os
import re
from typing import Any, Callable, Optional  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from os_brick import exception
from os_brick import executor
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
@staticmethod
@utils.retry(retry=utils.retry_if_exit_code, retry_param=139, interval=0.5, backoff_rate=0.5)
def get_lv_info(root_helper: str, vg_name: Optional[str]=None, lv_name: Optional[str]=None) -> list[dict[str, Any]]:
    """Retrieve info about LVs (all, in a VG, or a single LV).

        :param root_helper: root_helper to use for execute
        :param vg_name: optional, gathers info for only the specified VG
        :param lv_name: optional, gathers info for only the specified LV
        :returns: List of Dictionaries with LV info

        """
    cmd = LVM.LVM_CMD_PREFIX + ['lvs', '--noheadings', '--unit=g', '-o', 'vg_name,name,size', '--nosuffix']
    if lv_name is not None and vg_name is not None:
        cmd.append('%s/%s' % (vg_name, lv_name))
    elif vg_name is not None:
        cmd.append(vg_name)
    try:
        out, _err = priv_rootwrap.execute(*cmd, root_helper=root_helper, run_as_root=True)
    except putils.ProcessExecutionError as err:
        with excutils.save_and_reraise_exception(reraise=True) as ctx:
            if 'not found' in err.stderr or 'Failed to find' in err.stderr:
                ctx.reraise = False
                LOG.info('Logical Volume not found when querying LVM info. (vg_name=%(vg)s, lv_name=%(lv)s', {'vg': vg_name, 'lv': lv_name})
                out = None
    lv_list = []
    if out is not None:
        volumes = out.split()
        iterator = zip(*[iter(volumes)] * 3)
        for vg, name, size in iterator:
            lv_list.append({'vg': vg, 'name': name, 'size': size})
    return lv_list