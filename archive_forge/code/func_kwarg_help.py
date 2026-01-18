import mplfinance as mpf
import pandas as pd
import textwrap
def kwarg_help(func_name=None, kwarg_names=None, sort=False):
    func_kwarg_map = {'plot': mpf.plotting._valid_plot_kwargs, 'make_addplot': mpf.plotting._valid_addplot_kwargs, 'make_marketcolors': mpf._styles._valid_make_marketcolors_kwargs, 'make_mpf_style': mpf._styles._valid_make_mpf_style_kwargs, 'renko_params': mpf._utils._valid_renko_kwargs, 'pnf_params': mpf._utils._valid_pnf_kwargs, 'lines': mpf._utils._valid_lines_kwargs, 'scale_width_adjustment': mpf._widths._valid_scale_width_kwargs, 'update_width_config': mpf._widths._valid_update_width_kwargs}
    func_kwarg_aliases = {'addplot': mpf.plotting._valid_addplot_kwargs, 'marketcolors': mpf._styles._valid_make_marketcolors_kwargs, 'mpf_style': mpf._styles._valid_make_mpf_style_kwargs, 'style': mpf._styles._valid_make_mpf_style_kwargs, 'renko': mpf._utils._valid_renko_kwargs, 'pnf': mpf._utils._valid_pnf_kwargs, 'hlines': mpf._utils._valid_lines_kwargs, 'alines': mpf._utils._valid_lines_kwargs, 'tlines': mpf._utils._valid_lines_kwargs, 'vlines': mpf._utils._valid_lines_kwargs}
    if func_name is None:
        print('\nUsage: `kwarg_help(func_name)` or `kwarg_help(func_name,kwarg_names)`')
        print('        kwarg_help is available for the following func_names:')
        s = str(list(func_kwarg_map.keys()))
        text = textwrap.wrap(s, 68)
        for t in text:
            print('           ', t)
        print()
        return
    fkmap = {**func_kwarg_map, **func_kwarg_aliases}
    if func_name not in fkmap:
        raise ValueError('Function name "' + func_name + '" NOT a valid function name')
    if kwarg_names is not None and isinstance(kwarg_names, str):
        kwarg_names = [kwarg_names]
    if kwarg_names is not None and (not isinstance(kwarg_names, (list, tuple)) or not all([isinstance(k, str) for k in kwarg_names])):
        raise ValueError('kwarg_names must be a sequence (list,tuple) of strings')
    vks = fkmap[func_name]()
    df = pd.DataFrame(vks).T.drop('Validator', axis=1)
    df.index.name = 'Kwarg'
    if sort:
        df.sort_index(inplace=True)
    df.reset_index(inplace=True)
    if kwarg_names is not None:
        for k in kwarg_names:
            if k not in df['Kwarg'].values:
                print('   Warning: "' + k + '" is not a valid `kwarg_name` for `func_name` "' + func_name, '"')
        df = df[df['Kwarg'].isin(kwarg_names)]
        if len(df) < 1:
            raise ValueError(' None of specified `kwarg_names` are valid for `func_name` "' + func_name, '"')
    df['Default'] = ["'" + d + "'" if isinstance(d, str) else str(d) for d in df['Default']]
    klen = df['Kwarg'].str.len().max() + 1
    dlen = df['Default'].str.len().max() + 1
    wraplen = max(40, 80 - (klen + dlen))
    df = df_wrapcols(df, wrap_columns={'Description': wraplen})
    dividers = []
    for col in df.columns:
        dividers.append('-' * int(df[col].str.len().max()))
    dfd = pd.DataFrame(dividers).T
    dfd.columns = df.columns
    dfd.index = pd.Index(['---'])
    df = pd.concat([dfd, df])
    formatters = {'Kwarg': make_left_formatter(klen), 'Default': make_left_formatter(dlen), 'Description': make_left_formatter(wraplen)}
    print('\n ', '-' * 78)
    print('  Kwargs for func_name "' + func_name + '":')
    s = df.to_string(formatters=formatters, index=False, justify='left')
    print('\n ', s.replace('\n', '\n  '))