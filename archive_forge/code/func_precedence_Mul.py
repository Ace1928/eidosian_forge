def precedence_Mul(item):
    if item.could_extract_minus_sign():
        return PRECEDENCE['Add']
    return PRECEDENCE['Mul']