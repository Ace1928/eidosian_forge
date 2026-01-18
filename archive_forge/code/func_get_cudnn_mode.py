import torch.cuda
def get_cudnn_mode(mode):
    if mode == 'RNN_RELU':
        return int(_cudnn.RNNMode.rnn_relu)
    elif mode == 'RNN_TANH':
        return int(_cudnn.RNNMode.rnn_tanh)
    elif mode == 'LSTM':
        return int(_cudnn.RNNMode.lstm)
    elif mode == 'GRU':
        return int(_cudnn.RNNMode.gru)
    else:
        raise Exception(f'Unknown mode: {mode}')