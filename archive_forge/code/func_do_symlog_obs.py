def do_symlog_obs(observation_space, symlog_obs_user_setting):
    is_image_space = len(observation_space.shape) in [2, 3]
    return not is_image_space if symlog_obs_user_setting == 'auto' else symlog_obs_user_setting