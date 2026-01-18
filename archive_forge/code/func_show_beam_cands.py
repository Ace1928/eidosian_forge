from parlai.core.torch_agent import TorchAgent
from .controls import eval_attr
def show_beam_cands(n_best_beam_preds, history, dictionary):
    """
    Pretty-print the n-best candidates from beam search, along with their probabilities.

    Inputs:
      n_best_beam_preds: list length num_candidates of (prediction, score) pairs.
        prediction is a tensor of word indices, score is a single float tensor.
      history: ConvAI2History
      dictionary: parlai DictionaryAgent
    """
    print('')
    print('persona: ', history.persona_lines)
    print('partner_utts: ', history.partner_utts)
    print('own_utts: ', history.own_utts)
    print('')
    for idx, (pred, score) in enumerate(n_best_beam_preds):
        text = dictionary.vec2txt(pred.tolist())
        text = text.replace('__start__ ', '').replace(' __end__', '')
        print('%i  %.4f  %s' % (idx, score, text))
    print('')