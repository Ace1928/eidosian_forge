import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from ray.rllib.utils.framework import try_import_tf
def create_frozenlake_dream_image(dreamed_obs, dreamed_V, dreamed_a, dreamed_r_tp1, dreamed_ri_tp1, dreamed_c_tp1, value_target, initial_h, as_tensor=False):
    frozenlake_env.unwrapped.s = np.argmax(dreamed_obs, axis=0)
    rgb_array = frozenlake_env.render()
    image = Image.fromarray(rgb_array)
    draw_obj = ImageDraw.Draw(image)
    draw_obj.text((5, 6), f'Vt={dreamed_V:.2f} (Rt={value_target:.2f})', fill=(0, 0, 0))
    action_arrow = '<--' if dreamed_a == 0 else 'v' if dreamed_a == 1 else '-->' if dreamed_a == 2 else '^'
    draw_obj.text((5, 18), f'at={action_arrow} ({dreamed_a})', fill=(0, 0, 0))
    draw_obj.text((5, 30), f'rt+1={dreamed_r_tp1:.2f}', fill=(0, 0, 0))
    if dreamed_ri_tp1 is not None:
        draw_obj.text((5, 42), f'rit+1={dreamed_ri_tp1:.6f}', fill=(0, 0, 0))
    draw_obj.text((5, 54), f'ct+1={dreamed_c_tp1}', fill=(0, 0, 0))
    draw_obj.text((5, 66), f'|h|t={np.mean(np.abs(initial_h)):.5f}', fill=(0, 0, 0))
    np_img = np.asarray(image)
    if as_tensor:
        return tf.convert_to_tensor(np_img, dtype=tf.uint8)
    return np_img