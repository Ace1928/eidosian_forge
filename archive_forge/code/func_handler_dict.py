from collections import OrderedDict
import logging
import numpy as np
from minerl.herobraine.hero.spaces import MineRLSpace
import minerl.herobraine.hero.spaces as spaces
from typing import List, Any
import typing
from minerl.herobraine.hero.handler import Handler
@property
def handler_dict(self) -> typing.Dict[str, Handler]:
    return {h.to_string(): h for h in self.handlers}