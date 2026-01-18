def is_isometric_to_with_effort(A, B, return_isometries=False, tries=10):
    A = improved_triangulation(A, tries=tries)
    B = improved_triangulation(B, tries=tries)
    try:
        return A.is_isometric_to(B, return_isometries=return_isometries)
    except RuntimeError:
        return []