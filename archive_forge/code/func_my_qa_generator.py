import os, re
from rwkv.model import RWKV  #### pip install rwkv --upgrade
from rwkv.utils import PIPELINE, PIPELINE_ARGS
def my_qa_generator(ctx):
    out_tokens = []
    out_len = 0
    out_str = ''
    occurrence = {}
    state = None
    for i in range(999):
        if i == 0:
            out, state = pipeline.model.forward(pipeline.encode(ctx), state)
        else:
            out, state = pipeline.model.forward([token], state)
        for n in occurrence:
            out[n] -= 0.4 + occurrence[n] * 0.4
        token = pipeline.sample_logits(out, temperature=1.0, top_p=0.2)
        if token == 0:
            break
        out_tokens += [token]
        for n in occurrence:
            occurrence[n] *= 0.996
        occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
        tmp = pipeline.decode(out_tokens[out_len:])
        if '�' not in tmp and (not tmp.endswith('\n')):
            out_str += tmp
            print(tmp, end='', flush=True)
            out_len = i + 1
        elif '\n\n' in tmp:
            tmp = tmp.rstrip()
            out_str += tmp
            print(tmp, end='', flush=True)
            break
    return out_str.strip()