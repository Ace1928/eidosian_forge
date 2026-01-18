from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.handler import Handler
from typing import List
import gym
import minerl.herobraine
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.hero.handlers import TranslationHandler
from minerl.herobraine.hero.mc import ALL_ITEMS, INVERSE_KEYMAP
from minerl.herobraine.env_spec import EnvSpec
from collections import OrderedDict
def create_agent_handlers(self) -> List[Handler]:
    return []