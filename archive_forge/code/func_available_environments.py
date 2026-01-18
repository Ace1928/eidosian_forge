from openchat.agents.blender import BlenderGenerationAgent
from openchat.agents.dialogpt import DialoGPTAgent
from openchat.agents.dodecathlon import DodecathlonAgent
from openchat.agents.gptneo import GPTNeoAgent
from openchat.agents.safety import OffensiveAgent, SensitiveAgent
from openchat.agents.reddit import RedditAgent
from openchat.agents.unlikelihood import UnlikelihoodAgent
from openchat.agents.wow import WizardOfWikipediaGenerationAgent
from openchat.envs.interactive import InteractiveEnvironment
from openchat.utils.terminal_utils import draw_openchat
@staticmethod
def available_environments():
    return ['interactive']