import cirq
def test_align_left_subset_of_operations():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    tag = 'op_to_align'
    c_orig = cirq.Circuit([cirq.Moment([cirq.Y(q1)]), cirq.Moment([cirq.X(q2)]), cirq.Moment([cirq.X(q1).with_tags(tag)]), cirq.Moment([cirq.Y(q2)]), cirq.measure(*[q1, q2], key='a')])
    c_exp = cirq.Circuit([cirq.Moment([cirq.Y(q1)]), cirq.Moment([cirq.X(q1).with_tags(tag), cirq.X(q2)]), cirq.Moment(), cirq.Moment([cirq.Y(q2)]), cirq.measure(*[q1, q2], key='a')])
    cirq.testing.assert_same_circuits(cirq.toggle_tags(cirq.align_left(cirq.toggle_tags(c_orig, [tag]), context=cirq.TransformerContext(tags_to_ignore=[tag])), [tag]), c_exp)