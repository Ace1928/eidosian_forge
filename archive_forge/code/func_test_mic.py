def test_mic():
    import ase
    import numpy as np
    tol = 1e-09
    cell = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.0]]) * 10
    pos = np.dot(np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2], [0.25, 0.5, 0.0]]), cell)
    a = ase.Atoms('C4', pos, cell=cell, pbc=True)
    rpos = a.get_scaled_positions()
    d01F = np.linalg.norm(np.dot(rpos[1], cell))
    d01T = np.linalg.norm(np.dot(rpos[1] - np.array([0, 1, 0]), cell))
    d02F = np.linalg.norm(np.dot(rpos[2], cell))
    d02T = d02F
    d03F = np.linalg.norm(np.dot(rpos[3], cell))
    d03T = np.linalg.norm(np.dot(rpos[3] - np.array([0, 1, 0]), cell))
    assert abs(a.get_distance(0, 1, mic=False) - d01F) < tol
    assert abs(a.get_distance(0, 2, mic=False) - d02F) < tol
    assert abs(a.get_distance(0, 3, mic=False) - d03F) < tol
    assert abs(a.get_distance(0, 1, mic=True) - d01T) < tol
    assert abs(a.get_distance(0, 2, mic=True) - d02T) < tol
    assert abs(a.get_distance(0, 3, mic=True) - d03T) < tol
    assert all(abs(a.get_distance(0, 1, mic=False, vector=True) - np.array([7.5, np.sqrt(18.75), 5.0])) < tol)
    assert all(abs(a.get_distance(0, 2, mic=False, vector=True) - np.array([3.0, np.sqrt(3.0), 2.0])) < tol)
    assert np.all(abs(a.get_distance(0, 1, mic=True, vector=True) - np.array([-2.5, np.sqrt(18.75), -5.0])) < tol)
    assert np.all(abs(a.get_distance(0, 2, mic=True, vector=True) - np.array([3.0, np.sqrt(3.0), 2.0])) < tol)
    all_dist = a.get_all_distances(mic=False)
    assert abs(all_dist[0, 1] - d01F) < tol
    assert abs(all_dist[0, 2] - d02F) < tol
    assert abs(all_dist[0, 3] - d03F) < tol
    assert all(abs(np.diagonal(all_dist)) < tol)
    all_dist_mic = a.get_all_distances(mic=True)
    assert abs(all_dist_mic[0, 1] - d01T) < tol
    assert abs(all_dist_mic[0, 2] - d02T) < tol
    assert abs(all_dist_mic[0, 3] - d03T) < tol
    assert all(abs(np.diagonal(all_dist)) < tol)
    for i in range(4):
        assert all(abs(a.get_distances(i, [0, 1, 2, 3], mic=False) - all_dist[i]) < tol)
    assert all(abs(a.get_distances(0, [0, 1, 2, 3], mic=True) - all_dist_mic[0]) < tol)
    assert all(abs(a.get_distances(1, [0, 1, 2, 3], mic=True) - all_dist_mic[1]) < tol)
    assert all(abs(a.get_distances(2, [0, 1, 2, 3], mic=True) - all_dist_mic[2]) < tol)
    assert all(abs(a.get_distances(3, [0, 1, 2, 3], mic=True) - all_dist_mic[3]) < tol)
    assert np.all(abs(a.get_distances(0, [0, 1, 2, 3], mic=False, vector=True) - np.array([a.get_distance(0, i, vector=True) for i in [0, 1, 2, 3]])) < tol)
    assert np.all(abs(a.get_distances(1, [0, 1, 2, 3], mic=False, vector=True) - np.array([a.get_distance(1, i, vector=True) for i in [0, 1, 2, 3]])) < tol)
    assert np.all(abs(a.get_distances(2, [0, 1, 2, 3], mic=False, vector=True) - np.array([a.get_distance(2, i, vector=True) for i in [0, 1, 2, 3]])) < tol)
    assert np.all(abs(a.get_distances(3, [0, 1, 2, 3], mic=False, vector=True) - np.array([a.get_distance(3, i, vector=True) for i in [0, 1, 2, 3]])) < tol)
    assert np.all(abs(a.get_distances(0, [0, 1, 2, 3], mic=True, vector=True) - np.array([a.get_distance(0, i, mic=True, vector=True) for i in [0, 1, 2, 3]])) < tol)
    assert np.all(abs(a.get_distances(1, [0, 1, 2, 3], mic=True, vector=True) - np.array([a.get_distance(1, i, mic=True, vector=True) for i in [0, 1, 2, 3]])) < tol)
    assert np.all(abs(a.get_distances(2, [0, 1, 2, 3], mic=True, vector=True) - np.array([a.get_distance(2, i, mic=True, vector=True) for i in [0, 1, 2, 3]])) < tol)
    assert np.all(abs(a.get_distances(3, [0, 1, 2, 3], mic=True, vector=True) - np.array([a.get_distance(3, i, mic=True, vector=True) for i in [0, 1, 2, 3]])) < tol)
    a.set_distance(0, 1, 11.0, mic=False)
    assert abs(a.get_distance(0, 1, mic=False) - 11.0) < tol
    assert abs(a.get_distance(0, 1, mic=True) - np.sqrt(46)) < tol
    a.set_distance(0, 1, 3.0, mic=True)
    assert abs(a.get_distance(0, 1, mic=True) - 3.0) < tol