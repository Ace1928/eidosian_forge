import os
import subprocess
import io
import time
import threading
import Pyro4
def run_env():
    try:
        env = gym.make('MineRLNavigateDense-v0')
    except Exception as e:
        print('Pyro traceback:')
        print(''.join(Pyro4.util.getPyroTraceback()))
        raise e
    for _ in range(3):
        env.reset()
        for _ in range(100):
            env.step(env.action_space.no_op())