def log_input_scalar(name, data, run=None):
    run.config[name] = data
    wandb.termlog(f'Setting config: {name} to {data}')