import torch.cuda
def init_dropout_state(dropout, train, dropout_seed, dropout_state):
    dropout_desc_name = 'desc_' + str(torch.cuda.current_device())
    dropout_p = dropout if train else 0
    if dropout_desc_name not in dropout_state or dropout_state[dropout_desc_name].get() is None:
        if dropout_p == 0:
            dropout_state[dropout_desc_name] = Unserializable(None)
        else:
            dropout_state[dropout_desc_name] = Unserializable(torch._cudnn_init_dropout_state(dropout_p, train, dropout_seed, self_ty=torch.uint8, device=torch.device('cuda')))
    dropout_ts = dropout_state[dropout_desc_name].get()
    return dropout_ts