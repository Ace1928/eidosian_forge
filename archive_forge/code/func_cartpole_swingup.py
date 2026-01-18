from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv
def cartpole_swingup(from_pixels=True, height=64, width=64, frame_skip=2, channels_first=True):
    return DMCEnv('cartpole', 'swingup', from_pixels=from_pixels, height=height, width=width, frame_skip=frame_skip, channels_first=channels_first)