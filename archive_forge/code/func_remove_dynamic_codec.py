from collections import namedtuple
import codecs
@staticmethod
def remove_dynamic_codec(name):
    name = name.lower()
    if name in RL_Codecs.__rl_dynamic_codecs:
        RL_Codecs.__rl_codecs_data.pop(name, None)
        RL_Codecs.__rl_codecs_cache.pop(name, None)
        RL_Codecs.__rl_dynamic_codecs.remove(name)