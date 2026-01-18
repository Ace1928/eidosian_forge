from collections import namedtuple
import codecs
@staticmethod
def reset_dynamic_codecs():
    map(RL_Codecs.remove_dynamic_codec, RL_Codecs.__rl_dynamic_codecs)