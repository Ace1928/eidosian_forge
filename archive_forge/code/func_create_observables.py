import gym
from minerl.env import _fake, _singleagent
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.mc import ALL_ITEMS
from typing import List
def create_observables(self) -> List[Handler]:
    return [handlers.POVObservation(self.resolution), handlers.FlatInventoryObservation(ALL_ITEMS)]