from nibabel.cifti2 import cifti2_axes
def test_name_conversion():
    """
    Tests the automatic name conversion to a format recognized by CIFTI-2
    """
    func = cifti2_axes.BrainModelAxis.to_cifti_brain_structure_name
    for base_name, input_names in equivalents:
        assert base_name == func(base_name)
        for name in input_names:
            assert base_name == func(name)