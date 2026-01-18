import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def remove_dead(blocks, args, func_ir, typemap=None, alias_map=None, arg_aliases=None):
    """dead code elimination using liveness and CFG info.
    Returns True if something has been removed, or False if nothing is removed.
    """
    cfg = compute_cfg_from_blocks(blocks)
    usedefs = compute_use_defs(blocks)
    live_map = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)
    call_table, _ = get_call_table(blocks)
    if alias_map is None or arg_aliases is None:
        alias_map, arg_aliases = find_potential_aliases(blocks, args, typemap, func_ir)
    if config.DEBUG_ARRAY_OPT >= 1:
        print('args:', args)
        print('alias map:', alias_map)
        print('arg_aliases:', arg_aliases)
        print('live_map:', live_map)
        print('usemap:', usedefs.usemap)
        print('defmap:', usedefs.defmap)
    alias_set = set(alias_map.keys())
    removed = False
    for label, block in blocks.items():
        lives = {v.name for v in block.terminator.list_vars()}
        if config.DEBUG_ARRAY_OPT >= 2:
            print('remove_dead processing block', label, lives)
        for out_blk, _data in cfg.successors(label):
            if config.DEBUG_ARRAY_OPT >= 2:
                print('succ live_map', out_blk, live_map[out_blk])
            lives |= live_map[out_blk]
        removed |= remove_dead_block(block, lives, call_table, arg_aliases, alias_map, alias_set, func_ir, typemap)
    return removed