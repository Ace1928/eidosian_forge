import traceback
from copy import deepcopy
import json
import logging
from minerl.env.comms import retry
from minerl.env.exceptions import MissionInitException
import os
from minerl.herobraine.wrapper import EnvWrapper
import struct
from minerl.env.malmo import InstanceManager, MinecraftInstance, launch_queue_logger_thread, malmo_version
import uuid
import coloredlogs
import gym
import socket
import time
from lxml import etree
from minerl.env import comms
import xmltodict
from concurrent.futures import ThreadPoolExecutor
import cv2
from minerl.herobraine.env_spec import EnvSpec
from typing import Any, Callable, Dict, List, Optional, Tuple
def noop_action(self):
    """Gets the no-op action for the environment.

        In addition one can get the no-op/default action directly from the action space.

            env.action_space.noop()


        Returns:
            Any: A no-op action in the env's space.
        """
    return self.action_space.no_op()