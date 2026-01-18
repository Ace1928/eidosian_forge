def random_unit_vector():
    while 1:
        v = rng.uniform(-1, 1, 3)
        norm = np.linalg.norm(v)
        if norm > eps:
            return v / norm