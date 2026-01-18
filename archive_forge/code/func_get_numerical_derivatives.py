def get_numerical_derivatives(positions, mode, epsilon):
    if mode == 'distance':
        mode_n = 0
    elif mode == 'angle':
        mode_n = 1
    elif mode == 'dihedral':
        mode_n = 2
    derivs = np.zeros((2 + mode_n, 3))
    for i in range(2 + mode_n):
        for j in range(3):
            pos = positions.copy()
            pos[i, j] -= epsilon
            if mode == 'distance':
                minus = np.linalg.norm(pos[1] - pos[0])
            elif mode == 'angle':
                minus = get_angles([pos[0] - pos[1]], [pos[2] - pos[1]])
            elif mode == 'dihedral':
                minus = get_dihedrals([pos[1] - pos[0]], [pos[2] - pos[1]], [pos[3] - pos[2]])
            pos[i, j] += 2 * epsilon
            if mode == 'distance':
                plus = np.linalg.norm(pos[1] - pos[0])
            elif mode == 'angle':
                plus = get_angles([pos[0] - pos[1]], [pos[2] - pos[1]])
            elif mode == 'dihedral':
                plus = get_dihedrals([pos[1] - pos[0]], [pos[2] - pos[1]], [pos[3] - pos[2]])
            derivs[i, j] = (plus - minus) / (2 * epsilon)
    return derivs