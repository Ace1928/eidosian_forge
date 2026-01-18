def compute_epsilon(RF):
    return RF(0.5) ** (RF.prec() // 2)