def plot_kwargs_cb(idx, labels=None):
    kwargs = {'ls': ls[idx % len(ls)], 'c': c[idx % len(c)]}
    if labels:
        kwargs['label'] = labels[idx]
    return kwargs