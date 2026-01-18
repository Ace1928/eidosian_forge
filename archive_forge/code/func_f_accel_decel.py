def f_accel_decel(t, old_d, new_d, abruptness=1, soonness=1.0):
    """
    abruptness
      negative abruptness (>-1): speed up down up
      zero abruptness : no effect
      positive abruptness: speed down up down
      
    soonness
      for positive abruptness, determines how soon the
      speedup occurs (0<soonness < inf)
    """
    a = 1.0 + abruptness

    def _f(t):
        f1 = lambda t: 0.5 ** (1 - a) * t ** a
        f2 = lambda t: 1 - f1(1 - t)
        return (t < 0.5) * f1(t) + (t >= 0.5) * f2(t)
    return old_d * _f((t / new_d) ** soonness)