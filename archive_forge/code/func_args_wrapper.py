import time
def args_wrapper(*args):
    """Generates callback arguments for model.fit()
    for a set of callback objects.
    Callback objects like PandasLogger(), LiveLearningCurve()
    get passed in.  This assembles all their callback arguments.
    """
    out = defaultdict(list)
    for callback in args:
        callback_args = callback.callback_args()
        for k, v in callback_args.items():
            out[k].append(v)
    return dict(out)