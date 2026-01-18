import abc
from abc import ABC
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero import handlers as H, mc
from minerl.herobraine.hero.mc import ALL_ITEMS, INVERSE_KEYMAP, SIMPLE_KEYBOARD_ACTION
from minerl.herobraine.env_spec import EnvSpec
from typing import List
import numpy as np
def create_agent_start(self) -> List[Handler]:
    gui_handler = H.GuiScale(np.random.uniform(*self.guiscale_range))
    gamma_handler = H.GammaSetting(np.random.uniform(*self.gamma_range))
    fov_handler = H.FOVSetting(np.random.uniform(*self.fov_range))
    cursor_size_handler = H.FakeCursorSize(np.random.randint(self.cursor_size_range[0], self.cursor_size_range[1] + 1))
    return [gui_handler, gamma_handler, fov_handler, cursor_size_handler]