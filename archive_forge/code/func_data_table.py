def data_table(fn_list, datalabel, keyatom):
    """Generate a data table from a list of input xpk files.

    Parameters
    ----------
    fn_list : list
        List of .xpk file names.
    datalabel : str
        The data element reported.
    keyatom : str
        The name of the nucleus used as an index for the data table.

    Returns
    -------
    outlist : list
       List of table rows indexed by ``keyatom``.

    """
    outlist = []
    dict_list, label_line_list = _read_dicts(fn_list, keyatom)
    minr = dict_list[0]['minres']
    maxr = dict_list[0]['maxres']
    for dictionary in dict_list:
        if maxr < dictionary['maxres']:
            maxr = dictionary['maxres']
        if minr > dictionary['minres']:
            minr = dictionary['minres']
    res = minr
    while res <= maxr:
        count = 0
        key = str(res)
        line = key
        for dictionary in dict_list:
            label = label_line_list[count]
            if key in dictionary:
                line = line + '\t' + XpkEntry(dictionary[key][0], label).fields[datalabel]
            else:
                line += '\t*'
            count += 1
        line += '\n'
        outlist.append(line)
        res += 1
    return outlist