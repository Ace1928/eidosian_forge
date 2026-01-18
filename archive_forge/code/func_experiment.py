import argparse
import ray
from ray import train, tune
import ray.rllib.algorithms.ppo as ppo
def experiment(config):
    iterations = config.pop('train-iterations')
    algo = ppo.PPO(config=config)
    checkpoint = None
    train_results = {}
    for i in range(iterations):
        train_results = algo.train()
        if i % 2 == 0 or i == iterations - 1:
            checkpoint = algo.save(train.get_context().get_trial_dir())
        train.report(train_results)
    algo.stop()
    config['num_workers'] = 0
    eval_algo = ppo.PPO(config=config)
    eval_algo.restore(checkpoint)
    env = eval_algo.workers.local_worker().env
    obs, info = env.reset()
    done = False
    eval_results = {'eval_reward': 0, 'eval_eps_length': 0}
    while not done:
        action = eval_algo.compute_single_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        eval_results['eval_reward'] += reward
        eval_results['eval_eps_length'] += 1
    results = {**train_results, **eval_results}
    train.report(results)