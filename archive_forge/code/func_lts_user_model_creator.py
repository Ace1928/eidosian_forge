from recsim import choice_model
from recsim.environments import (
from ray.rllib.env.wrappers.recsim import make_recsim_env
from ray.tune import register_env
def lts_user_model_creator(env_ctx):
    return lts.LTSUserModel(env_ctx['slate_size'], user_state_ctor=lts.LTSUserState, response_model_ctor=lts.LTSResponse)