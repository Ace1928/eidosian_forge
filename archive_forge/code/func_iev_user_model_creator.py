from recsim import choice_model
from recsim.environments import (
from ray.rllib.env.wrappers.recsim import make_recsim_env
from ray.tune import register_env
def iev_user_model_creator(env_ctx):
    return iev.IEvUserModel(env_ctx['slate_size'], choice_model_ctor=choice_model.MultinomialProportionalChoiceModel, response_model_ctor=iev.IEvResponse, user_state_ctor=iev.IEvUserState, seed=env_ctx['seed'])