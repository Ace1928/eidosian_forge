def get_link_back_to_kubeflow():
    wandb_kubeflow_url = os.getenv('WANDB_KUBEFLOW_URL')
    return f'{wandb_kubeflow_url}/#/runs/details/{{workflow.uid}}'