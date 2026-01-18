def annotation_params_for_line(shape_type, shape_args, position):
    x0 = shape_args['x0']
    x1 = shape_args['x1']
    y0 = shape_args['y0']
    y1 = shape_args['y1']
    X = [x0, x1]
    Y = [y0, y1]
    R = 'right'
    T = 'top'
    L = 'left'
    C = 'center'
    B = 'bottom'
    M = 'middle'
    aY = max(Y)
    iY = min(Y)
    eY = _mean(Y)
    aaY = _argmax(Y)
    aiY = _argmin(Y)
    aX = max(X)
    iX = min(X)
    eX = _mean(X)
    aaX = _argmax(X)
    aiX = _argmin(X)
    position, pos_str = _prepare_position(position)
    if shape_type == 'vline':
        if position == set(['top', 'left']):
            return _df_anno(R, T, X[aaY], aY)
        if position == set(['top', 'right']):
            return _df_anno(L, T, X[aaY], aY)
        if position == set(['top']):
            return _df_anno(C, B, X[aaY], aY)
        if position == set(['bottom', 'left']):
            return _df_anno(R, B, X[aiY], iY)
        if position == set(['bottom', 'right']):
            return _df_anno(L, B, X[aiY], iY)
        if position == set(['bottom']):
            return _df_anno(C, T, X[aiY], iY)
        if position == set(['left']):
            return _df_anno(R, M, eX, eY)
        if position == set(['right']):
            return _df_anno(L, M, eX, eY)
    elif shape_type == 'hline':
        if position == set(['top', 'left']):
            return _df_anno(L, B, iX, Y[aiX])
        if position == set(['top', 'right']):
            return _df_anno(R, B, aX, Y[aaX])
        if position == set(['top']):
            return _df_anno(C, B, eX, eY)
        if position == set(['bottom', 'left']):
            return _df_anno(L, T, iX, Y[aiX])
        if position == set(['bottom', 'right']):
            return _df_anno(R, T, aX, Y[aaX])
        if position == set(['bottom']):
            return _df_anno(C, T, eX, eY)
        if position == set(['left']):
            return _df_anno(R, M, iX, Y[aiX])
        if position == set(['right']):
            return _df_anno(L, M, aX, Y[aaX])
    raise ValueError('Invalid annotation position "%s"' % (pos_str,))