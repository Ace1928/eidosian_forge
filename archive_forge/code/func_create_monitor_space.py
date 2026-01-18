from abc import abstractmethod
import types
from minerl.herobraine.hero.handlers.translation import TranslationHandler
import typing
from minerl.herobraine.hero.spaces import Dict
from minerl.herobraine.hero.handler import Handler
from typing import List
import jinja2
import gym
from lxml import etree
import os
import abc
import importlib
from minerl.herobraine.hero import spaces
def create_monitor_space(self):
    return self._singlify(spaces.Dict({agent: spaces.Dict({m.to_string(): m.space for m in self.monitors}) for agent in self.agent_names}))