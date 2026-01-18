import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def human_mitosis():
    """Image of human cells undergoing mitosis.

    Returns
    -------
    human_mitosis: (512, 512) uint8 ndarray
        Data of human cells undergoing mitosis taken during the preparation
        of the manuscript in [1]_.

    Notes
    -----
    Copyright David Root. Licensed under CC-0 [2]_.

    References
    ----------
    .. [1] Moffat J, Grueneberg DA, Yang X, Kim SY, Kloepfer AM, Hinkle G,
           Piqani B, Eisenhaure TM, Luo B, Grenier JK, Carpenter AE, Foo SY,
           Stewart SA, Stockwell BR, Hacohen N, Hahn WC, Lander ES,
           Sabatini DM, Root DE (2006) A lentiviral RNAi library for human and
           mouse genes applied to an arrayed viral high-content screen. Cell,
           124(6):1283-98 / :DOI: `10.1016/j.cell.2006.01.040` PMID 16564017

    .. [2] GitHub licensing discussion
           https://github.com/CellProfiler/examples/issues/41

    """
    return _load('data/mitosis.tif')